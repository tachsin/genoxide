// serde for arrays of any length `M` (serde's own implementations stop at 32), as tuples like
// serde's, for `#[serde(with = "...")]`

use serde::de::{Deserialize, Deserializer, Error, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeTuple, Serializer};
use std::fmt;
use std::marker::PhantomData;

// an array to serialize
struct ArrayRef<'a, T, const M: usize>(&'a [T; M]);

impl<T: Serialize, const M: usize> Serialize for ArrayRef<'_, T, M> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tuple = serializer.serialize_tuple(M)?;
        for element in self.0 {
            tuple.serialize_element(element)?;
        }
        tuple.end()
    }
}

// a deserialized array
struct ArrayOwned<T, const M: usize>([T; M]);

impl<'de, T: Deserialize<'de>, const M: usize> Deserialize<'de> for ArrayOwned<T, M> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ArrayVisitor<T, const M: usize>(PhantomData<T>);

        impl<'de, T: Deserialize<'de>, const M: usize> Visitor<'de> for ArrayVisitor<T, M> {
            type Value = ArrayOwned<T, M>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "an array of length {M}")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut elements = Vec::with_capacity(M);
                while elements.len() < M {
                    match seq.next_element()? {
                        Some(element) => elements.push(element),
                        None => return Err(A::Error::invalid_length(elements.len(), &self)),
                    }
                }
                match elements.try_into() {
                    Ok(array) => Ok(ArrayOwned(array)),
                    Err(elements) => Err(A::Error::invalid_length(elements.len(), &self)),
                }
            }
        }

        deserializer.deserialize_tuple(M, ArrayVisitor::<T, M>(PhantomData))
    }
}

// `[T; M]`
pub(crate) mod array {
    use super::*;

    pub(crate) fn serialize<S, T, const M: usize>(
        array: &[T; M],
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize,
    {
        ArrayRef(array).serialize(serializer)
    }

    pub(crate) fn deserialize<'de, D, T, const M: usize>(
        deserializer: D,
    ) -> Result<[T; M], D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de>,
    {
        ArrayOwned::deserialize(deserializer).map(|array| array.0)
    }
}

// `Vec<[T; M]>`
pub(crate) mod vec_of_arrays {
    use super::*;

    pub(crate) fn serialize<S, T, const M: usize>(
        arrays: &[[T; M]],
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize,
    {
        serializer.collect_seq(arrays.iter().map(ArrayRef))
    }

    pub(crate) fn deserialize<'de, D, T, const M: usize>(
        deserializer: D,
    ) -> Result<Vec<[T; M]>, D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de>,
    {
        let arrays = Vec::<ArrayOwned<T, M>>::deserialize(deserializer)?;
        Ok(arrays.into_iter().map(|array| array.0).collect())
    }
}

// `Option<[[T; M]; M]>`
pub(crate) mod option_matrix {
    use super::*;

    // a matrix to serialize, row by row
    struct MatrixRef<'a, T, const M: usize>(&'a [[T; M]; M]);

    impl<T: Serialize, const M: usize> Serialize for MatrixRef<'_, T, M> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut tuple = serializer.serialize_tuple(M)?;
            for row in self.0 {
                tuple.serialize_element(&ArrayRef(row))?;
            }
            tuple.end()
        }
    }

    pub(crate) fn serialize<S, T, const M: usize>(
        matrix: &Option<[[T; M]; M]>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize,
    {
        match matrix {
            Some(matrix) => serializer.serialize_some(&MatrixRef(matrix)),
            None => serializer.serialize_none(),
        }
    }

    pub(crate) fn deserialize<'de, D, T, const M: usize>(
        deserializer: D,
    ) -> Result<Option<[[T; M]; M]>, D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de>,
    {
        let matrix = Option::<ArrayOwned<ArrayOwned<T, M>, M>>::deserialize(deserializer)?;
        Ok(matrix.map(|matrix| matrix.0.map(|row| row.0)))
    }
}
